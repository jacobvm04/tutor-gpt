import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { useState } from "react";
import { useRouter } from "next/navigation";
import Swal from 'sweetalert2'

export default function Forgot(props: any) {
  const [password, setPassword] = useState('');

  const supabase = createClientComponentClient()
  const router = useRouter()

  const handleReset = async (e: any) => {
    e.preventDefault()
    const { data, error } = await supabase.auth.updateUser({ password })

    if (error) {
      alert("There was an error updating your password.")
      await Swal.fire({
        title: 'Error!',
        text: "Something went wrong",
        icon: "error",
        confirmButtonText: "Close"
      })
      console.error(error)
      return
    }

    if (data) {
      await Swal.fire({
        title: "Success!",
        text: "Password updated successfully",
        icon: "success",
        confirmButtonText: "Close"
      })
      router.push(`/`)
    }
  }

  return (
    <form action="#" className="mt-8 grid grid-cols-6 gap-6">

      <div className="col-span-6">
        <label
          htmlFor="Password"
          className="block text-sm font-medium text-gray-700"
        >
          Password
        </label>

        <input
          type="password"
          id="Password"
          name="password"
          className="p-2 mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>

      <div className="col-span-6 sm:flex sm:items-center sm:gap-4">
        <button
          className="inline-block shrink-0 rounded-md border border-neon-green bg-neon-green px-12 py-3 text-sm font-medium transition hover:bg-transparent hover:text-blue-600 focus:outline-none focus:ring active:text-blue-500"
          onClick={handleReset}
        >
          Reset
        </button>

      </div>
    </form>
  )
}
