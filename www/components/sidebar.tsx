import { useRouter } from "next/navigation"
import { useState } from 'react'
import { FaEdit, FaTrash } from "react-icons/fa"
import { GrClose } from "react-icons/gr"
import { Session } from '@supabase/supabase-js'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'

interface Conversation {
  conversation_id: string;
  name: string;
}

const URL = process.env.NEXT_PUBLIC_URL;

export default function Sidebar({
  conversations,
  authSession,
  currentConversation,
  setCurrentConversation,
  setConversations,
  newChat,
  userId,
  isSidebarOpen,
  setIsSidebarOpen
}: {
  conversations: Array<Conversation>
  authSession: Session | null
  currentConversation: Conversation
  setCurrentConversation: Function
  setConversations: Function
  newChat: Function
  userId: string
  isSidebarOpen: boolean
  setIsSidebarOpen: Function
}) {
  // const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const router = useRouter()
  const supabase = createClientComponentClient()

  async function handleSignOut() {
    await supabase.auth.signOut()
    console.log("Signed out")
    location.reload()
  }

  async function editConversation(cur: Conversation) {
    const newName = prompt("Enter a new name for the conversation")
    if (!newName)
      return
    fetch(`${URL}/api/conversations/update`, {
      method: "POST",
      body: JSON.stringify({
        conversation_id: cur.conversation_id,
        name: newName
      }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((data) => {
        const copy = { ...cur }
        copy.name = newName
        setConversations(conversations.map(conversation =>
          conversation.conversation_id === copy.conversation_id ? copy : conversation
        ))
      })
  }

  async function deleteConversation(conversation: Conversation) {
    const check = confirm("Are you sure you want to delete this conversation, this action is irreversible?")
    if (!check)
      return
    const { conversation_id } = conversation
    await fetch(`${URL}/api/conversations/delete?user_id=${userId}&conversation_id=${conversation_id}`)
      .then((res) => res.json())
    // Delete the conversation_id from the conversations state variable
    setConversations(conversations.filter(cur => cur.conversation_id !== conversation_id));

    // If it was the currentConversation, change the currentConversation to the next one in the list
    if (conversation === currentConversation) {
      if (conversations.length > 1) {
        setCurrentConversation(conversations[0]);
        console.log("Current Conversation", currentConversation)
      } else {
        // If there is no current conversation create a new one
        const newConversationId = await newChat();
        setCurrentConversation(newConversationId);
        console.log("Current Conversation", currentConversation)
        setConversations([newConversationId]);
      }
    }

  }

  async function addChat() {
    const conversationId = await newChat();
    const newConversation: Conversation = {
      name: "Untitled",
      conversation_id: conversationId
    }

    setConversations([...conversations, newConversation])
  }

  return (
    <div className={`fixed lg:static z-20 inset-0 flex-none h-full w-full lg:absolute lg:h-auto lg:overflow-visible lg:pt-0 lg:w-60 xl:w-72 lg:block ${isSidebarOpen ? "" : "hidden"}`}>
      <div className={`h-full scrollbar-trigger overflow-hidden bg-white sm:w-3/5 w-4/5 lg:w-full flex flex-col ${isSidebarOpen ? "fixed lg:static" : "sticky"} top-0 left-0`}>
        {/* Section 1: Top buttons */}
        <div className="flex justify-between items-center p-4 gap-2 border-b border-gray-300">
          <button className="bg-neon-green rounded-lg px-4 py-2 w-full lg:w-full h-10" onClick={addChat}>New Chat</button>
          <button className="lg:hidden bg-neon-green rounded-lg px-4 py-2 h-10" onClick={() => setIsSidebarOpen(false)}><GrClose /></button>
        </div>

        {/* Section 2: Scrollable items */}
        <div className="flex flex-col flex-1 overflow-y-auto divide-y divide-gray-300">
          {/* Replace this with your dynamic items */}
          {conversations.map((cur, i) => (
            <div key={i} className={`flex justify-between items-center p-4 cursor-pointer hover:bg-gray-200 ${currentConversation === cur ? "bg-gray-200" : ""}`} onClick={() => setCurrentConversation(cur)}>
              <div>
                <h2 className="font-bold overflow-ellipsis overflow-hidden">{cur.name || "Untitled"}</h2>
              </div>
              <div className="flex flex-row justify-end gap-2 items-center w-1/5">
                <button className="text-gray-500" onClick={() => editConversation(cur)}>
                  <FaEdit />
                </button>
                <button className="text-red-500" onClick={() => deleteConversation(cur)}>
                  <FaTrash />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Section 3: Authentication information */}
        <div className="border-t border-gray-300 p-4 w-full">
          {/* Replace this with your authentication information */}
          {authSession ?
            (<button className="bg-neon-green rounded-lg px-4 py-2 w-full" onClick={handleSignOut}>Sign Out</button>) :
            (<button className="bg-neon-green rounded-lg px-4 py-2 w-full" onClick={() => router.push("/auth")}>Sign In</button>)}
        </div>
      </div>
    </div>
  )
}
